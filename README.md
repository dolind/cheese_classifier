---
title: A Cheese Classifier
sdk: gradio
app_file: app_cloud/app.py
requirements_file: app_cloud/requirements.txt
---


# A Cheese Classifier

Serves as playground app for fast ai image inference

## Cloud app
There is a cloud app hosted on  https://huggingface.co/spaces/dolind/whichcheese.

The models must be added with git-lfs as huggingface has a 10MB file size limit.

## Javascript app
In addition there is a javascript app, which can be run at:

https://www.storymelange.com/cheese_classifier/

## Git Troubles with GIT-LFS
To run with github pages the model must be added to git without git lfs and in binary mode.
Check `.gitattributes`.

Github has a 100MB file size limit.

To avoid git issues, i work on a local branch `hugging_face_main` and push to huggingface with `git push space hugging_face_main:main`.

Where `space` is the remote on hugging face.

In the future this could surely be automated via Github actions.