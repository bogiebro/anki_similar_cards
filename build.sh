#!/bin/sh
cd src
rm -rf __pycache__
cat > manifest.json <<HERE
{
    "name": "Anki Similar Notes",
    "package": "anki_similar_notes",
    "mod": $(date +'%s')
}
HERE
zip -r ../anki_similar_notes.ankiaddon *
rm manifest.json
