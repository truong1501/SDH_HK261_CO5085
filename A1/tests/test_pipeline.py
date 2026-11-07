import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset
from src.config import build_run_config
from src.data import validate_splits
from src.models import build_model, set_frozen_bn_eval
from src.train import feature_cache_key, run_standard, run_freeze_all, check_budget, BudgetExhausted
from src.server import queue, pending


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 101)
    def get_classifier(self):
        return self.fc
    def forward_features(self, x):
        return x.mean((2, 3))
    def forward_head(self, x, pre_logits=False):
        return x if pre_logits else self.fc(x)
    def forward(self, x):
        return self.forward_head(self.forward_features(x))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_budget_and_priority_queue(self):
        import time
        with self.assertRaises(BudgetExhausted):
            check_budget(SimpleNamespace(deadline=time.monotonic()+1))
        check_budget(SimpleNamespace())
        self.assertEqual(len(queue()), 20)
        self.assertEqual(len(set(queue())), 20)
        self.assertTrue(all(seed == 42 for _, seed in queue()[:8]))
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/"R2_s42";folder.mkdir()
            (folder/"metrics.json").write_text("{}")
            self.assertNotIn(("R2",42), pending(temp))
            self.assertEqual(len(pending(temp)),19)

    def test_invalid_split_and_config(self):
        validate_splits([0,1], [2], 3)
        for train,val in [([0,1],[1,2]), ([0,0],[1,2]), ([0],[1])]:
            with self.assertRaises(ValueError): validate_splits(train,val,3)
        with self.assertRaises(ValueError): build_run_config('R1',42,batch_size=0)
        self.assertTrue(any('batch_size' in n for n in build_run_config('R1',42,batch_size=8).notes))

    def test_actual_models_forward_backward_and_freeze(self):
        for key,run in [('resnet50','R3'),('deit_s','R7')]:
            mode=build_run_config(run,42).mode.copy()
            mode['pretrained']=False
            model=build_model(key,mode)
            model.train(); set_frozen_bn_eval(model)
            x=torch.randn(2,3,224,224)
            output=model(x)
            self.assertEqual(tuple(output.shape),(2,101))
            output.square().mean().backward()
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
            self.assertTrue(any(p.grad is not None for p in model.parameters() if p.requires_grad))
            for module in model.modules():
                if isinstance(module,nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
                    self.assertFalse(module.training)
            del model

    def test_train_resume_and_frozen_feature_path(self):
        dataset=TensorDataset(torch.rand(8,3,16,16),torch.arange(8))
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            split=temp/'splits';split.mkdir()
            for name,values in [('train_idx',[0,1]),('val_idx',[2]),('classes',['a','b'])]:
                (split/f'{name}.json').write_text(json.dumps(values))
            args=SimpleNamespace(out_root=str(temp/'runs'),data_root='synthetic',splits_dir=str(split),num_workers=0)
            device=torch.device('cpu')
            with patch('src.train.build_model',side_effect=lambda *a,**k:TinyModel()), patch('src.train.build_datasets',return_value=(dataset,dataset,dataset)):
                cfg=build_run_config('R4',42,epochs=1,batch_size=4)
                out=temp/'standard';out.mkdir()
                run_standard(cfg,args,device,None,str(out))
                checkpoint=torch.load(out/'last.pt',weights_only=True)
                self.assertEqual(checkpoint['epoch'],1)
                cfg.mode['epochs']=2
                run_standard(cfg,args,device,None,str(out))
                self.assertEqual(torch.load(out/'last.pt',weights_only=True)['epoch'],2)
                frozen=temp/'frozen';frozen.mkdir()
                cfg=build_run_config('R2',42,epochs=1,batch_size=3)
                run_freeze_all(cfg,args,device,None,str(frozen))
                self.assertTrue((frozen/'metrics.json').is_file())
                key=feature_cache_key(cfg,args)
                (split/'val_idx.json').write_text('[3]')
                self.assertNotEqual(key,feature_cache_key(cfg,args))


if __name__=='__main__':
    unittest.main()
