# Thesis Project Progress

This section tracks the progress of the project. The following table shows the progress made each week:

| Week | Progress |
|----------|----------|
| 1  | Read papers on previous related work  |
| 2  | Worked on the code to download the street view images that will be used to model nature visibility in urban environments.<br>I created a Jupyter notebook that retrieves the road network of the selected city using OSMnx, selects points on the roads, retrieves features for each point, and downloads corresponding images. The notebook includes four main sections: <br><ol><li>Retrieving the road network and saving it as a GeoPackage file</li><li>Selecting points on the road edges</li><li>Downloading features for each point and matching them to each point</li><li>Downloading images for each point and saving them locally</li></ol>|
| 3  | I completed the following tasks:<br><ol><li>Added image segmentation using facebook/mask2former-swin-large-cityscapes-semanic. This enables the code to segment street images into differet regions based on their content. By leveraging this segmentation, we can subsequently identify the road centers and calculate the GVI (Green View Index) value. </li><li>Implemented code to find the center of the roads in street images following the approach proposed by Matthew Danis. This allows for the classification of images as usable or unusable based on the presence of a road centre.</li> |
| 4  | My focus was on cropping the panoramic images to ensure a consistent approach across the dataset. Moreover, I wrote the necessary code to calculate the GVI for each point and stored the results in a GeoDataFrame. |
| 5  | I helped Yúri in modifying the code to enable the calculation of GVI for a group of address points. Moreover, I divided the pipeline into separate components, allowing independent execution of point sampling, image downloading, and image processing. Furthermore, I initiated the execution of the code in not only Kampala, Uganda but also in Amsterdam, Netherlands. |


I will update this README file regularly to reflect my progress.