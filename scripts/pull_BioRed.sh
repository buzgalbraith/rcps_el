#!/bib/bash 
# pull the BioRed dataset 
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip
# move it to the data directory 
mkdir -p ~/.data/
unzip BIORED.zip -d ~/.data/
# clean up unneeded files
rm BIORED.zip