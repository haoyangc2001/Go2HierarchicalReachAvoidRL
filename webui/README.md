# Project Web UI

This directory contains the static project website for HD-MCRA safe quadruped navigation.

## Local Preview

Serve this directory with any static file server, then open index.html in a browser. The only local asset required by the page is assets/agiro-demo.mp4.

## GitHub Pages

The repository includes .github/workflows/deploy-webui.yml, which publishes this directory on pushes to the master branch that modify webui/.

In the GitHub repository settings, open **Pages** and select **GitHub Actions** as the source. After the workflow completes, the project site will be available from the repository's GitHub Pages URL.
