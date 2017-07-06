from flask import render_template, redirect, url_for, Flask
from forms import FilepathForm
from flask_wtf.csrf import CsrfProtect
import sys
import os.path

app = Flask(__name__)

@app.route('/input', methods=["GET", "POST"])
def input():
    form = FilepathForm()
    return render_template('index.html', form=form)

app.secret_key = os.path.expandvars("FLASK_SECRET_KEY")
