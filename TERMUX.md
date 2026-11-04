# Publish from Termux
pkg update -y && pkg install -y git unzip
termux-setup-storage
cd ~ && unzip ~/storage/downloads/aanpe-v4-master.zip -d aanpe && cd aanpe
git config --global user.name "Nikhil Chary Sriramoju"
git config --global user.email "sriramojunikhil66@gmail.com"
git init && git add . && git commit -m "AANPE v4.0"
git branch -M main
git remote add origin https://github.com/Nikhil-creat/aanpe.git
git push -u origin main
# Username: Nikhil-creat | Password: a GitHub Personal Access Token (repo scope)
# Then: repo Settings > Pages > Deploy from a branch > main / (root)
# Update later: git add . && git commit -m "update" && git push

# Update after unzipping the new zip over the folder
cd ~/aanpe && unzip -o ~/storage/downloads/aanpe-v4-master.zip
git add . && git commit -m "Add CI, PWA, AI briefing" && git push
