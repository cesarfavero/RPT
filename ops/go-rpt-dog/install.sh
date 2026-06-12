#!/usr/bin/env sh
set -eu

cat <<'EOF'
RPT installer is temporarily disabled.

Reason:
- the current downloadable bundle still contains source-visible runtime files;
- public installation will resume only after the runtime is shipped as a compiled distribution.

Do not distribute the current Python runtime bundle from the public installer endpoint.

For approved private evaluation, use the controlled internal release process instead.
EOF

exit 1
