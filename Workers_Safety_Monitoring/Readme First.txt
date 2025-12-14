Object model creation

1.Arrange the files in this structure:
D:\PPE_KIT_DATASET\
    images\  --all your JPG/PNG images here
    labels\

2.Install and open LabelImg:
 terminal:
	pip install labelImg
	labelImg
3. Set YOLO format
	View > autosave
	(PascalVOC/YOLO) → select YOLO
	set image folder (images\)
	set label folder (labels\)
	set default label as object name (eg: ppekit)
4. W - Create RectBox
   Draw a box around the object in the image.
   Ctrl+S
   D - Next Image
5. Create data.yaml
	location:D:\PPE_KIT_DATASET\

       run data.yaml 
       run trainobject.py
6. The file is created in location:
	D:\runs\detect\safetyshoe_detector\weights\best.pt
