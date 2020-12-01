# Human Portrait Drawing

![](https://img.shields.io/static/v1?label=python&message=3.6|3.7&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=1.4&color=<COLOR>)

```
├── README.md
├── configs                         some config files          
│     ├── infer_config.json        
│     ├── model_config.json        
│     ├── server_config.json       
├── datasets    
│     ├── APDrawingDB               the data for trianing and validation          
├── face_detect                       
├── model_logs                      checkpoint and some log files  
├── modules
│     ├── data_trans.py             data augmentation
│     ├── dataloader.py                    
│     ├── losses.py                 
│     ├── metrics.py                miou
│     ├── networks.py               an implentation for u2net 
│     ├── trainer.py                custom class for model training  
├── samples                         networks for Unet, UnetPlus, Resnet50FCN, ResnetFPN
│     ├── inputs                    input samples  
│     ├── outputs                   model prediction for input samples
├── utils                    
├── train.py                        model training
├── inference.py                    model infernece
├── flask_server.py                 a simple flask api for human portrait drawing
├── server_test.py       

```
## data
- APDrawingDB

## trian
1.use default paramters 
```angular2
python train.py
```
2.use specific config file
```angular2
python train.py --config configs/model_config.json
```
3.resume checkpoint 
```angular2
python train.py --config config_path --resume checkpoint_path
```

## inference
best model(10.10.101.15): 
- /data/changqing/Human_Portrait_Pytorch/model_logs/U2NET/1125_210759/best_model.pth
> use default params
```angular2 
python inference.py
```     
> use specific config file
```angular2
python inference.py --path ./configs/infer_config.json   
```