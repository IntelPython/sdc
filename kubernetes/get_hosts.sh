# kubectl get statefulsets my-mpi-cluster-worker -o jsonpath='{.status.replicas}'

pod_names=$(kubectl get pod --selector=app=my-hpat-test,role=worker -o=jsonpath='{.items[*].metadata.name}')

if [ -f hostfile ]; then
  rm hostfile
fi


for p in ${pod_names}; do
  ip=$(kubectl get pod $p -o jsonpath='{.status.podIP}')
  echo $ip >> hostfile
done

for p in ${pod_names}; do
  kubectl cp hostfile $p:/root/
done
