import os
import argparse

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

def parseArgs(): 
    parser = argparse.ArgumentParser(description="Training ILT or Litho models")
    parser.add_argument("--model", "-m", default="GANOPC", type=str, help="Model Name: {GANOPC, NeuralILT, DAMOILT, CFNOILT, LithoGAN, DOINN, DAMOLitho, CFNOLitho}")
    parser.add_argument("--benchmark", "-s", default="MetalSet", type=str, help="Benchmark: {MatelSet, ViaSet}")
    parser.add_argument("--epochs", "-n", default=2, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", default=4, type=int, help="Batch size")
    parser.add_argument("--njobs", "-j", default=8, type=int, help="Number of jobs for DataLoader")
    parser.add_argument("--pretrain", "-p", default=True, type=bool, help="Pretrain or not")
    parser.add_argument("--load", "-l", default="", type=str, help="Load existing weights")
    parser.add_argument("--load_gen", "-g", default="", type=str, help="Load generator weights")
    parser.add_argument("--load_dis", "-d", default="", type=str, help="Load discriminator weights")
    parser.add_argument("--eval", "-e", default=False, type=bool, help="Evaluation only")
    parser.add_argument("--output", "-o", default="saved", type=str, help="Folder for saving the weights and images")
    return parser.parse_args()

if __name__ == "__main__": 
    args = parseArgs()
    
    Benchmark = args.benchmark
    ImageSize = (1024, 1024)
    if args.model in ["GANOPC", "LithoGAN"]: 
        ImageSize = (256, 256)
    elif args.model in ["NeuralILT", ]: 
        ImageSize = (512, 512)
    Epochs = args.epochs
    BatchSize = args.batch_size
    NJobs = args.njobs
    Pretrain = args.pretrain
    EvalOnly = args.eval
    Task = "Litho" if args.model in ["LithoGAN", "DOINN", "DAMOLitho", "CFNOLitho"] else "ILT"
    Folder = os.path.join(args.output, f"{args.benchmark}_{args.model}")
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
        assert False, f"[ERROR]: Unsupported model: {args.model}"
    
    train_loader, val_loader = None, None
    targets = None
    if Task == "Litho": 
        train_loader, val_loader = loadersLitho(Benchmark, ImageSize, BatchSize, NJobs)
    else: 
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
        model.evaluate(targets, finetune=False, folder=Folder)
    else: 
        model.evaluate(Benchmark, ImageSize, BatchSize, NJobs)
