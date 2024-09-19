import os

from IPython.lib import passwd

c = c  # noqa
c.NotebookApp.ip = "0.0.0.0"  # https://github.com/jupyter/notebook/issues/3946
c.NotebookApp.port = int(os.getenv("PORT", 8888))
c.NotebookApp.open_browser = False
c.NotebookApp.notebook_dir = "/wd"

# https://github.com/jupyter/notebook/issues/3130
c.FileContentsManager.delete_to_trash = False

# # sets a password if PASSWORD is set in the environment
# if "PASSWORD" in os.environ:
#     password = os.environ["PASSWORD"]
#     if password:
#         c.NotebookApp.password = passwd(password)
#     else:
#         c.NotebookApp.password = ""
#         c.NotebookApp.token = ""
#     del os.environ["PASSWORD"]
