if ! test -f /etc/hosts.bak; then
    sudo cp /etc/hosts /etc/hosts.bak
fi

sudo cp /etc/hosts.bak /etc/hosts

YAML_INFO="
`gcloud compute tpus tpu-vm list --zone=us-central1-f --format=yaml`
`gcloud compute tpus tpu-vm list --zone=europe-west4-a --format=yaml`
"

NAMES=(`echo "$YAML_INFO" | yq eval ".name" | cut -d "/" -f6 | grep -v "\-\-\-"`)
EXTERNAL_IPS=(`echo "$YAML_INFO" | yq eval ".networkEndpoints[0].accessConfig.externalIp" | grep -v "\-\-\-"`)

for index in ${!NAMES[*]}; do 
  echo "${EXTERNAL_IPS[$index]} ${NAMES[$index]}" | sudo tee -a /etc/hosts
done

echo "# last updated `date`" | sudo tee -a /etc/hosts

IP_ADDRESS=`hostname -I | awk '{print $1}'`
TPU_NAME=`echo "$YAML_INFO" | yq eval "select(.networkEndpoints[0].ipAddress==\"${IP_ADDRESS}\") | .name" | cut -d "/" -f6`
if [ -z "$TPU_NAME" ]; then
  echo "TPU_NAME not found for IP_ADDRESS=$IP_ADDRESS"
  exit 1
fi
sudo hostname $TPU_NAME