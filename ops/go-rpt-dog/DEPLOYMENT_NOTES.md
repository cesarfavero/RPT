# go.rpt.dog Deployment Notes

Purpose:

- operational note so the `go.rpt.dog` host details do not get lost across chats

Current deployment target:

- host: `38.242.192.239`
- ssh user: `rpt-go`

Operational reminder:

- `go.rpt.dog/install.sh` must stay disabled until the runtime is shipped as a compiled distribution
- do not restore public curl-to-install while the downloadable bundle still contains source-visible runtime files

Useful commands:

```bash
ssh rpt-go@38.242.192.239
curl -fsSL https://go.rpt.dog/install.sh | sed -n '1,120p'
```

Expected policy state:

- public site `rpt.dog` must not show signed URL examples
- public `go.rpt.dog/install.sh` must return the disabled installer message

If access details change, update this file immediately.
