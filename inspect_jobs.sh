job_ids=$(squeue -p MoE | awk '/[0-9]+/{print $1}')

tot_cpu_num=0

for job_id in $job_ids ; do
    echo "Job ID: $job_id"
    cpu_num=$(scontrol show job $job_id | grep -oP 'cpu=\K\d+' | cut -d'=' -f2)
    tot_cpu_num=$(($tot_cpu_num + $cpu_num))
done
echo $tot_cpu_num
