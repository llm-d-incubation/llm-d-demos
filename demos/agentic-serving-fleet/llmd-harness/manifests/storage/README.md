# Shared storage for the KV cache tier

`ENABLE_KV_STORAGE_TIER=1` needs a ReadWriteMany PVC named `KV_STORAGE_PVC` (default
`llm-d-kv-cache-storage`) in the GPU namespace. Two ways to get one:

**Existing Managed Lustre instance (existing instance):**

```bash
source config/env.sh
export LUSTRE_INSTANCE_PATH=projects/${PROJECT_ID}/locations/<zone>/instances/<name>
export LUSTRE_FILESYSTEM=<fs>     # from: gcloud lustre instances describe <name> --location <zone>
export LUSTRE_IP=<ip>             #   mountPoint is <ip>@tcp:/<fs>
export LUSTRE_CAPACITY=<capacity, e.g. 9000Gi>
export GPU_NAMESPACE=${NAMESPACE}-gpu
kubectl create namespace "${GPU_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
envsubst '${GPU_NAMESPACE} ${KV_STORAGE_PVC} ${LUSTRE_INSTANCE_PATH} ${LUSTRE_FILESYSTEM} ${LUSTRE_IP} ${LUSTRE_CAPACITY}' \
  < manifests/storage/lustre-kv-cache-pvc.yaml.tmpl | kubectl apply -f -
```

**Dynamic provisioning:** a `StorageClass` with provisioner `lustre.csi.storage.gke.io` (GKE ships
`lustre-rwx-{125,250,500,1000}mbps-per-tib` when the addon is enabled) and a plain RWX PVC. Creating an
instance takes a while and needs the VPC peering set up; the static path above is faster for a demo.

Rules that cost us time:

- Same region as the GPU nodes; cross-region storage is many times slower than recomputing
  (measured numbers in the harness README, "Shared-storage KV tier").
- One mount per Lustre filesystem name per node. A second PV for the same instance fails with
  `AlreadyExists ... same lustre filesystem name`; reuse the existing PVC instead.
- The Lustre CSI addon supports arm64 (GB200 / a4x) nodes.
