---
title: A Cheese Classifier
sdk: gradio
app_file: app_cloud/app.py
requirements_file: app_cloud/requirements.txt
---


# A Cheese Classifier

Serves as playground app for fast ai image inference

## Cloud app
There is a cloud app hosted on https://huggingface.co/spaces/dolind/whichcheese.

Git LFS must be used to add the models, due to Hugging Face’s 10MB file size limit.

## Javascript app
In addition, there is a JavaScript app available at:

https://www.storymelange.com/cheese_classifier/

## Git Troubles with GIT-LFS
To run with GitHub pages the model must be added to git without git lfs and in binary mode.
Check `.gitattributes`.

GitHub has a 100MB file size limit.

To avoid git issues, i work on a local branch `hugging_face_main` and push to huggingface with `git push space hugging_face_main:main`.

Where `space` is the remote on hugging face.

In the future, this could surely be automated via Github actions.