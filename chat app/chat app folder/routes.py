from flask import render_template, url_for, flash, redirect, request, abort
from pepperblog import app, db, bcrypt, mkd
from pepperblog.forms import LikeForm, RegistrationForm, ResetEmailForm,ResetPasswordForm, SortForm, LoginForm, UpdateAccountForm, NewPostForm
from pepperblog.models import User, Post, Like
from flask_login import login_user, logout_user,current_user, login_required, AnonymousUserMixin
import secrets
import os
from flask_sqlalchemy import BaseQuery
from PIL import Image


#export FLASK_APP=blog.py
#flask run
#export FLASK_DEBUG=1--debug mode update without exit





@app.route("/signup", methods=['GET','POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form=RegistrationForm()
    if form.validate_on_submit():
        hashed_pw=bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user=User(username=form.username.data,email=form.email.data,password=hashed_pw)
        db.session.add(user)
        db.session.commit()
        flash("Account created for {}. You can login now.".format(form.username.data),"success")
        return redirect(url_for('login'))
    return render_template("signup.html", title="Pepper: Sign Up", form=form)


@app.route("/login",methods=['GET','POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form=LoginForm()
    if form.validate_on_submit():
        user=User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password,form.password.data):
            login_user(user,remember=form.remember.data) #testpw=username.lower()
            page=request.args.get('next')
            if page:
                return redirect(page)
            else:
                return redirect(url_for('home'))

        else:
            flash("Login Unsuccessful for {}".format(form.email.data),"danger")
    return render_template("login.html", title="Pepper: Login", form=form)


