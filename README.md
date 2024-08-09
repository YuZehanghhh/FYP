# FYP

## Setup 

Install the necessary libraries
```bash
pip install torch torchvision
```

## Data
- [GAN testing datasets](https://drive.google.com/drive/folders/1RwCSaraEUctIwFgoQXWMKFvW07gM80_3) 
- Download the file and unzip it in `datasets/test` as following structure:
```

datasets
└── test					
      ├── progan	
      │── bigaan  	
      │── cyclegan
      │── deepfake
      │      .
      │      .
	  
```

- [Diffusion testing datasets](https://drive.google.com/file/d/1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t/view)
- Download the file and unzip it in `./diffusion_datasets` as following structure:
```

diffusion_datasets
└── dalle
└── glided
└── laion
└──     .
└──     .
		
 
	  
```

## Training

- [Training datasets](https://cmu.app.box.com/s/4syr4womrggfin0tsfhxohaec5dh6n48) (dataset size ~ 70GB). 

- Download and unzip the dataset in `datasets/train` directory. The overall structure should look like the following:
```
datasets
└── train			
      └── progan			
           ├── airplane
           │── bird
           │── boat
           │      .
           │      .
```
- The model can then be trained with the following command:
```bash
python train.py --name=clip_vitl14 --wang2020_data_path=datasets/ --data_mode=wang2020  --arch=CLIP:ViT-L/14  --fix_backbone
```

## Evaluation
 
- Evaluate the model on all the dataset:
```bash
python validate.py  --arch=CLIP:RN50  --ckpt=weights/cliprn50_parameters.pth   --result_folder=clip_rn50
```

- Evaluate the model on one generative model by specifying the paths of real and fake datasets:
```bash
python validate.py  --arch=CLIP:ViT-L/14   --ckpt=weights/clip14_parameters.pth   --result_folder=clip_vitl14  --real_path datasets/test/deepfake/0_real --fake_path datasets/test/deepfake/1_fake
```
- All the pretrained parameters are in 'weights' folder


