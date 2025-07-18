import os
import sys
import logging
import pandas as pd
from functools import partial
from sklearn.model_selection import StratifiedKFold
from peft import LoraConfig, get_peft_model
from transformers import AutoImageProcessor, SiglipForImageClassification

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms

from trainer import Trainer
from config import get_args
from lr_scheduler import get_sch
from data import DataSet, ImageFolder, get_transforms
from utils import seed_everything, handle_unhandled_exception, save_to_json
from create_valid import create_validation_folder

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed) #fix seed
    device = torch.device('cuda:0') #use cuda:0

    result_path = os.path.join(args.result_path, args.model +'_'+str(len(os.listdir(args.result_path))))
    os.makedirs(result_path)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))    
    logger.info(args)
    save_to_json(vars(args), os.path.join(result_path, 'config.json'))
    sys.excepthook = partial(handle_unhandled_exception,logger=logger)

    if not os.path.exists(os.path.join(args.data_path, 'valid')):
        create_validation_folder(args.data_path)
        logger.info('created validation folder')

    args.train_path = os.path.join(args.data_path, 'train')
    args.valid_path = os.path.join(args.data_path, 'valid')

    config = LoraConfig(
        r=args.r,
        lora_alpha=args.r,
        lora_dropout=0.1,
        target_modules=["k_proj", "q_proj"],
        bias="none",
        modules_to_save=["classifier"],
    )

    if args.continue_from is not None:
        model = SiglipForImageClassification.from_pretrained(args.continue_from, num_labels=393).to(device)
    else:
        model = SiglipForImageClassification.from_pretrained('google/siglip2-giant-opt-patch16-384', num_labels=393).to(device)
        
    print(model)
    model = get_peft_model(model, config)
    
    transform_train, transform_test = get_transforms() 

    train_dataset = ImageFolder(root=args.train_path, transform=transform_train)
    valid_dataset = ImageFolder(root=args.valid_path, transform=transform_test)
        
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_sch(args.scheduler, optimizer)

    if args.continue_from is not None:
        state = torch.load(os.path.join(args.continue_from, 'state.pt'))
        optimizer.load_state_dict(state['optimizer'])
        # scheduler.load_state_dict(state['scheduler'])
        epoch = state['epoch'] + 1
        print(f'starting from {epoch}-epoch')
    else:
        epoch = 0

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, #pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, #pin_memory=True
    )
    
    trainer = Trainer(
        train_loader, valid_loader, model, loss_fn, optimizer, scheduler, device, args.patience, args.epochs, result_path, logger, epoch=epoch)
    trainer.train() #start training

    # test_dataset = DataSet(file_list=test_data['path'].values, label=test_data['label'].values)
    # test_loader = DataLoader(
    #     test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    # ) #make test data loader

    # prediction[output_index] += trainer.test(test_loader) #softmax applied output; accumulate test prediction of current fold model
    # prediction.to_csv(os.path.join(result_path, 'sum.csv'), index=False) 
  