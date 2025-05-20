commit="update"
update="${1:-$commit}"
git add .
git commit -m "update"
git push origin master