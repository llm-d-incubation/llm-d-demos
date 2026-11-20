// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

// http_srv serves /var/lib/postgresql/data on :18888. deploy-patched-binaries.sh
// runs it inside postgres-0 so the ate-api Deployment and the atelet DaemonSet
// can download the patched binaries (bin_ateapi.gz, bin_atelet.gz) at pod start.
//
// SECURITY: this serves the WHOLE Postgres data directory, including the raw
// database files under pgdata/, unauthenticated, to anything that can reach the
// pod IP. It was a demo shortcut. For anything beyond a throwaway demo cluster,
// bake the binaries into images in Artifact Registry instead.
//
// Build (static, no module needed): CGO_ENABLED=0 go build -ldflags="-s -w" -o http_srv http_srv.go
package main

import "net/http"

func main() { http.ListenAndServe(":18888", http.FileServer(http.Dir("/var/lib/postgresql/data"))) }
