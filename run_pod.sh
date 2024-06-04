git push

gcloud compute tpus tpu-vm ssh $2 --worker=all --command="cd zett && git pull -f" --zone=us-central2-b
gcloud compute tpus tpu-vm ssh $2 --worker=all --command="cd zett && python3 train.py $1" --zone=us-central2-b