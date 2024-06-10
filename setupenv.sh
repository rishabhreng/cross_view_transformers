cd ~
git clone https://github.com/rishabhreng/cross_view_transformers.git
cd cross_view_transformers

conda create -y --name cvt python=3.8
conda activate cvt
conda install -y pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
pip install -e .
pip install --upgrade pytorch torchvision

echo Do you want to create the nuScenes dataset? y/n
read createAns
echo $createAns
if [ "$createAns" == "n" ];
then
  echo DONE
elif ["$createAns" == "y"];
then
# Define the URLs to download
cd ~
mkdir raw/ && cd raw/

wget "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval01_keyframes.tgz" &
wget  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval02_keyframes.tgz" &
wget  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval03_keyframes.tgz" &
wget  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval04_keyframes.tgz" &
wget  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval05_keyframes.tgz" &
wget  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval06_keyframes.tgz" &
wget  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_keyframes.tgz" &
wget  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval08_keyframes.tgz" &
wget  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval09_keyframes.tgz" &
wget  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval10_keyframes.tgz" &
wget  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz" &
wget  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz" &
wget  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/nuScenes-map-expansion-v1.3.zip" &
wget  "https://www.cs.utexas.edu/~bzhou/cvt/cvt_labels_nuscenes.tar.gz" 
wait

echo "Downloads completed."

cd ~
tar -xzvf /cvt_labels_nuscenes.tar.gz -C media/datasets
echo "Labels extracted"

mkdir media/datasets/nuscenes/

# Untar all the keyframes and metadata
for f in $(ls raw/v1.0-*.tgz); do tar -xzvf $f -C /media/datasets/nuscenes; done

# Map expansion must go into the maps folder
conda -y install unzip

unzip raw/nuScenes-map-expansion-v1.3.zip -d media/datasets/nuscenes/maps

else
  echo "bad input"
fi