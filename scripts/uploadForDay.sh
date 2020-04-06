#!/bin/bash
source /home/pi/allsky/config.sh
source /home/pi/allsky/scripts/filename.sh

cd  /home/pi/allsky/scripts

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

if [ $# -lt 1 ]
  then
    echo -en "${RED}You need to pass a day argument\n"
        echo -en "    ex: uploadForDay.sh 20180119${NC}\n"
        exit 3
fi

# Upload keogram
echo -e "Uploading Keogram\n"
KEOGRAM="/home/pi/allsky/images/$1/keogram/keogram-$1.jpg"
if [[ $PROTOCOL == "ssh" ]] ; then
  scp $KEOGRAM $USER@$HOST:$KEOGRAM_DIR/
else
  lftp "$PROTOCOL"://"$USER":"$PASSWORD"@"$HOST":"$KEOGRAM_DIR" -e "set net:max-retries 1; put $KEOGRAM; bye" -u "$USER","$PASSWORD"
fi
echo -e "\n"

# Upload Startrails
echo -e "Uploading Startrails\n"
STARTRAILS="/home/pi/allsky/images/$1/startrails/startrails-$1.jpg"
if [[ $PROTOCOL == "ssh" ]] ; then
  scp $STARTRAILS $USER@$HOST:$STARTRAILS_DIR/
else
  lftp "$PROTOCOL"://"$USER":"$PASSWORD"@"$HOST":"$STARTRAILS_DIR" -e "set net:max-retries 1; put $STARTRAILS; bye"
fi
echo -e "\n"

# Upload timelapse
echo -e "Uploading Timelapse\n"
TIMELAPSE="/home/pi/allsky/images/$1/allsky-$1.mp4"
if [[ $PROTOCOL == "ssh" ]] ; then
  scp $TIMELAPSE $USER@$HOST:$MP4DIR/
else
  lftp "$PROTOCOL"://"$USER":"$PASSWORD"@"$HOST":"$MP4DIR" -e "set net:max-retries 1; put $TIMELAPSE; bye"
fi
echo -e "\n"
