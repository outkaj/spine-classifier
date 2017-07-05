from flask import render_template, redirect, url_for

from . import app
from .forms import FilepathForm

@app.route('/input', methods=["GET", "POST"])
def login():
    form = FilepathForm()
    if form.validate_on_submit():
        return redirect(url_for('index'))
    return render_template('index.html', form=form)
