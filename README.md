# spectraClassification
Python repository for analysing spectra of SDSS objects.

## Current stage
The full dataset of spectra (about 4M) should be downloaded by now on the lofar machine.  
Currently I am looking at ways to identify the spectra and match them to the actual objects.  


## Updates
A script has been devised to match the spectra downloaded with the previous photometric data of the SDSS.  




## Bad news
It currently seems as the two databases we have do not match.  
The min distance (computed using formula for ra and dec) is tipically on the order of 0.1  
Script check_min_length was devised to see what is the inherent mean distance of one object to its closest neighbour. Results:  
mean is: 0.029994536113678666
std is: 0.016524732056337615
