#!/bin/bash
source /home/pi/allsky/config.sh

# TODO Needs fixing when civil twilight happens after midnight
cd /home/pi/allsky/scripts

latitude=60.7N
longitude=135.02W
timezone=-0700
streamDaytime=false

if [[ $DAYTIME == "1" ]] ; then
	streamDaytime=true;
fi

echo "Posting Next Twilight Time"
today=`date +%Y-%m-%d`
time="$(sunwait list set civil $latitude $longitude)"
timeNoZone=${time:0:5}
echo { > data.json
echo \"sunset\": \"$today"T"$timeNoZone":00.000$timezone"\", >> data.json
echo \"streamDaytime\": \"$streamDaytime\" >> data.json
echo } >> data.json
echo "Uploading data.json"
if [[ $PROTOCOL == "ssh" ]] ; then
	scp data.json $USER@$HOST:$IMGDIR
else
	lftp "$PROTOCOL"://"$USER":"$PASSWORD"@"$HOST":"$IMGDIR" -e "set net:max-retries 1; set net:timeout 20; put data.json; bye"
fi
