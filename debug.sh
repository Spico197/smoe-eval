# ids=(
#     "1667527"
#     "1667528"
#     "1667529"
#     "1667530"
#     "1667531"
#     "1667532"
# )

# for job_id in "${ids[@]}"; do
#     content=$(ls logs/ | grep $job_id | xargs -I {} grep "|acc_norm|" logs/{})
#     echo "$job_id: $content"
#     # if [ -z $content ]; then
#     #     echo $job_id
#     # fi
# done
