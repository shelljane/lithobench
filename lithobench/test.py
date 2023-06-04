import os
import argparse
from importlib import import_module

from model import *
from dataset import *

from ilt.ganopc import GANOPC
from ilt.neuralilt import NeuralILT
from ilt.damoilt import DAMOILT
from ilt.cfnoilt import CFNOILT
from litho.lithogan import LithoGAN
from litho.doinn import DOINN
from litho.damolitho import DAMOLitho
from litho.cfnolitho import CFNOLitho

# Example: python3 lithobench/test.py -m lithobench/ilt/neuralilt.py -a NeuralILT -i 512 -t ILT -o dev -s MetalSet -l saved/MetalSet_NeuralILT/net.pth
def parseArgs(): 
    parser = argparse.ArgumentParser(description="Training ILT or Litho models")
    parser.add_argument("--model", "-m", default="GANOPC", type=str, help="Model Name: {GANOPC, NeuralILT, DAMOILT, CFNOILT, LithoGAN, DOINN, DAMOLitho, CFNOLitho} or a user-provided filename")
    parser.add_argument("--alias", "-a", default=None, type=str, help="Model alias, required when the model is provided by the user")
    parser.add_argument("--task", "-t", default=None, type=str, help="Task: {ILT, Litho}")
    parser.add_argument("--img_size", "-i", default=None, type=int, help="User-defined image size, required when the model is provided by the user")
    parser.add_argument("--benchmark", "-s", default="MetalSet", type=str, help="Benchmark: {MatelSet, ViaSet, StdMetal, StdContact}")
    parser.add_argument("--epochs", "-n", default=2, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", default=4, type=int, help="Batch size")
    parser.add_argument("--njobs", "-j", default=8, type=int, help="Number of jobs for DataLoader")
    parser.add_argument("--pretrain", "-p", default=False, type=bool, help="Pretrain or not")
    parser.add_argument("--load", "-l", default="", type=str, help="Load existing weights")
    parser.add_argument("--load_gen", "-g", default="", type=str, help="Load generator weights")
    parser.add_argument("--load_dis", "-d", default="", type=str, help="Load discriminator weights")
    parser.add_argument("--eval", "-e", default=True, type=bool, help="Evaluation only")
    parser.add_argument("--output", "-o", default="saved", type=str, help="Folder for saving the weights and images")
    parser.add_argument("--shots", "-c", default=False, action='store_true', help="Count the shots (slow) or not")
    return parser.parse_args()

if __name__ == "__main__": 
    args = parseArgs()
    
    Benchmark = args.benchmark
    ImageSize = (1024, 1024)
    if args.model in ["GANOPC", "LithoGAN"]: 
        ImageSize = (256, 256)
    elif args.model in ["NeuralILT", ]: 
        ImageSize = (512, 512)
    if not args.img_size is None: 
        ImageSize = (args.img_size, args.img_size)
    Epochs = args.epochs
    BatchSize = args.batch_size
    NJobs = args.njobs
    Pretrain = args.pretrain
    EvalOnly = args.eval
    Task = "Litho" if args.model in ["LithoGAN", "DOINN", "DAMOLitho", "CFNOLitho"] else "ILT"
    if not args.task is None: 
        Task = args.task
    alias = args.model
    if not args.alias is None: 
        alias = args.alias
    Folder = os.path.join(args.output, f"{args.benchmark}_{alias}")
    if not os.path.exists(Folder): 
        os.mkdir(Folder)
    Filenames = None
    if args.model in ["GANOPC", "DAMOILT", "LithoGAN", "DAMOLitho"] and (len(args.load_gen) > 0 and len(args.load_dis) > 0): 
        Filenames = [args.load_gen, args.load_dis]
    elif len(args.load) > 0: 
        Filenames = args.load
    ToSave = None
    if args.model in ["GANOPC", "DAMOILT", "LithoGAN", "DAMOLitho"]: 
        ToSave = [f"{Folder}/netG.pth", f"{Folder}/netD.pth"]
    else: 
        ToSave = f"{Folder}/net.pth"
    
    model = None
    if args.model == "GANOPC": 
        model = GANOPC(size=ImageSize)
    elif args.model == "NeuralILT": 
        model = NeuralILT(size=ImageSize)
    elif args.model == "DAMOILT": 
        model = DAMOILT(size=ImageSize)
    elif args.model == "CFNOILT": 
        model = CFNOILT(size=ImageSize)
    elif args.model == "LithoGAN": 
        model = LithoGAN(size=ImageSize)
    elif args.model == "DOINN": 
        model = DOINN(size=ImageSize)
    elif args.model == "DAMOLitho": 
        model = DAMOLitho(size=ImageSize)
    elif args.model == "CFNOLitho": 
        model = CFNOLitho(size=ImageSize)
    else: 
        if not os.path.exists(args.model): 
            assert False, f"[ERROR]: Unsupported model: {args.model}"
        modulename = ".".join(args.model[:-3].split("/"))
        module = import_module(modulename)
        MyModel = getattr(module, alias)
        model = MyModel(size=ImageSize)
        
    if Benchmark in ["MetalSet", "ViaSet"]: 
        train_loader, val_loader = None, None
        targets = None
        if Task == "Litho": 
            train_loader, val_loader = loadersLitho(Benchmark, ImageSize, BatchSize, NJobs)
        else: # ILT
            train_loader, val_loader = loadersILT(Benchmark, ImageSize, BatchSize, NJobs)
            targets = evaluate.getTargets(samples=None, dataset=Benchmark)
        
        if not Filenames is None: 
            model.load(Filenames)
        if Pretrain and not EvalOnly: 
            model.pretrain(train_loader, val_loader, epochs=Epochs)
            model.save(ToSave)
        if not EvalOnly: 
            model.train(train_loader, val_loader, epochs=Epochs)
            model.save(ToSave)
        if Task == "ILT": 
            model.evaluate(targets, finetune=True, folder=Folder, shot=args.shots)
        else: # Litho
            model.evaluate(Benchmark, ImageSize, BatchSize, NJobs, folder=Folder, samples=10)
    elif Benchmark == "StdMetal": 
        if not Filenames is None: 
            model.load(Filenames)
        if Task == "ILT": 
            targets = evaluate.getTargets(samples=None, dataset=Benchmark)
            model.evaluate(targets, finetune=True, folder=Folder, shot=args.shots)
        else: # Litho
            loader = loadersAllLitho(Benchmark, (2048, 2048), BatchSize, NJobs)
            model.evaluate(Benchmark, ImageSize, BatchSize, NJobs, test_loader=loader, folder=Folder, samples=10)
    elif Benchmark == "StdContact": 
        train_loader, val_loader = None, None
        targets = None
        if Task == "Litho": 
            train_loader, val_loader = loadersLitho(Benchmark, ImageSize, BatchSize, NJobs, drop_last=True)
        else: # ILT
            train_loader, val_loader = loadersILT(Benchmark, ImageSize, BatchSize, NJobs, drop_last=True)
            targets = evaluate.getTargets(samples=None, dataset=Benchmark)
        
        if not Filenames is None: 
            model.load(Filenames)
        if Pretrain: 
            model.pretrain(train_loader, val_loader, epochs=Epochs)
        model.train(train_loader, val_loader, epochs=Epochs)
        if Task == "ILT": 
            model.evaluate(targets, finetune=True, folder=Folder, shot=args.shots)
        else: # Litho
            loader = loadersAllLitho("StdContactTest", (2048, 2048), BatchSize, NJobs)
            model.evaluate(Benchmark, ImageSize, BatchSize, NJobs, test_loader=loader, folder=Folder, samples=10)
    else: 
        assert False, f"[ERROR]: Unsupported benchmark {Benchmark}"