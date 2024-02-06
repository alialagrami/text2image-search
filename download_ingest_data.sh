mkdir -p data
cd data
wget https://storage.googleapis.com/ads-dataset/subfolder-0.zip -O file.zip
mkdir -p images_data
unzip file.zip -d images_data
cd ..
source venv/bin/activate
python ingest