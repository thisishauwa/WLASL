# Setting Up Your Own GitHub Repository

To push this code to your own GitHub repository:

1. **Create a new repository on GitHub**
   - Go to https://github.com/new
   - Name your repository (e.g., "WLASL" or "sign-language-recognition")
   - Choose public or private access as preferred
   - Do not initialize with README, .gitignore, or license (since we already have a local repository)
   - Click "Create repository"

2. **Update the remote in your local repository**
   Run the following commands, replacing `YOUR_USERNAME` with your GitHub username and `YOUR_REPO_NAME` with the name you chose:

   ```bash
   git remote rename origin upstream
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   ```

3. **Stage and commit your changes**
   ```bash
   git add .gitignore code/I3D/simple_inference.py code/I3D/test_i3d_cpu.py download_weights.md requirements.txt
   git commit -m "Update .gitignore and add inference scripts"
   ```

4. **Push to your repository**
   ```bash
   git push -u origin master
   ```

After completing these steps, your code will be available in your own GitHub repository.
