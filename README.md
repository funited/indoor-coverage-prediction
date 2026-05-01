# Indoor-Coverage-Prediction
Coverage (Heatmap) prediction for complex indoor environments using machine learning methods is shared in this project.

The indoor environment is defined by its fundamental structure, such as the arrangement of walls, windows, and doorways, alongside varying configurations of furniture placement, an Attention U-Net with efficient networks as the backbone is introduced to predict signal propagation in environments with a variety of objects, effectively simulating the diverse range of furniture typically found in indoor spaces. 

Ray-based 3D wireless simulation is performed on [Remcom](https://www.remcom.com/)'s [Wireless InSite®](https://www.remcom.com/wireless-insite-propagation-software)

***Setup of the indoor environment in Wireless InSite®***: 
<img width="354" height="177" alt="image" src="https://github.com/user-attachments/assets/22027aca-2076-4a5c-97e9-ac4aab1654b0" />

***Model Structure***: 
<img width="687" height="250" alt="image" src="https://github.com/user-attachments/assets/f4d780e7-d539-4902-a3a7-88d37e7324cf" />

***Results***:
<img width="712" height="802" alt="image" src="https://github.com/user-attachments/assets/ec4d5849-c0e0-452b-9f47-5c6b6700d747" />

# Usage
You can try data collected at either 5 GHz or 28 GHz, templates can be found in [notebook_usage](https://github.com/funited/indoor-coverage-prediction/tree/main/notebook_usage). 

Set "split = High" to work with receiver grid at 40 inches (1.06m) above the ground; "split = Low" to work with receiver grid at 30 inches (0.76m) above the ground.

You can find the dataset on Hugging Face: https://huggingface.co/datasets/funited/Indoor_Coverage_Prediction

For those who prefer "copy & paste" please check out the link in 'Project on Drive".
# License
MIT

# Reference
Please cite [Transfer Learning and Double U-Net Empowered Wave Propagation Model in Complex Indoor Environment](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=z1chur8AAAAJ&citation_for_view=z1chur8AAAAJ:d1gkVwhDpl0C) in your work if this project helps.

# Key Contributors
<img width="229" height="58" alt="image" src="https://github.com/user-attachments/assets/ac95ebca-1937-43d4-9375-893af20be85b" />

<img width="189" height="69" alt="image" src="https://github.com/user-attachments/assets/d09a4dbd-dd45-410a-9e48-94bd4748c65b" />



# Feedback
For questions and comments, please feel free to contact fun*ited(outlook.com) (remove *)
