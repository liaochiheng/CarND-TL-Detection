# CarND-TL-Detection
Traffic light detection used for the final CarND project.


## Installation
* `conda env create -f envionment.yml`

## Directory looks like:
<CarND-TL-Detection>  
+ data  
  + Bosch_Small_TrafficLight_Dataset  
    + rgb  
    - bosch_label_map.pbtxt  
    - train.yaml  
    ...  
  + dataset_rgb  
+ models  
  + ssd_mobilnet_...  
  + ...  
+ src  
...  

## Run script to generate tf-record
* `cd src`
* `python create_bosch_tf_record.py`
* You will see a file 'rgb.record' located in 'data/Bosch_Small_TrafficLight_Dataset/'