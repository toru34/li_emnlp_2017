# Download
FILE_ID=0B6N7tANPyVeBNmlSX19Ld2xDU1E
FILE_NAME=summary.tar.gz
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}

# Unpack
tar -zxvf summary.tar.gz
gzip -d ./sumdata/train/*.gz

# Move
mkdir data
mv ./sumdata/train/* ./data/
