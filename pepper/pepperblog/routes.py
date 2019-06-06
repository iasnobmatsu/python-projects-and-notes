from flask import render_template, url_for, flash, redirect, request
from pepperblog import app, db, bcrypt
from pepperblog.forms import RegistrationForm, LoginForm, UpdateAccountForm, NewPostForm
from pepperblog.models import User, Post
from flask_login import login_user, logout_user,current_user, login_required
import secrets
import os
from PIL import Image

#export FLASK_APP=blog.py
#flask run
#export FLASK_DEBUG=1--debug mode update without exit

#dummy data
posts=[
    {
        "user":"Ramon Cajal",
        "title":"Purkinje Cell Structure",
        "content":"Mitch Prinstein is the John Van Seters Distinguished Professor of Psychology and Neuroscience, and a member of the Clinical Psychology Program. Mitch’s research uses a developmental psychopathology framework to understand how adolescents’ interpersonal experiences, particularly among peers,  are associated with depression, self-injury, and health risk behaviors.  Mitch’s work has two areas of focus.",
        "date":"May 31, 1880"
    },{
        "user":"Mitch Prinstein",
        "title":"Peer Relations",
        "content":"rrrrrrrrrr\nr\n\n\n\n\n\rrr",
        "date":"August 2, 2019"
    }
]


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html',posts=posts, title="Pepper: Blog, Share, More")


@app.route("/explore")
def explore():
    return render_template('explore.html', title="Pepper:Explore", posts=posts)


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


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('login'))


def saveProfile(pic):
    rad=secrets.token_hex(8)
    _,file_ext=os.path.splitext(pic.filename)
    picture_to_save=rad+file_ext
    picture_path=os.path.join(app.root_path,"static/profiles/", picture_to_save)
    pic_image=Image.open(pic)
    pic_image.thumbnail((180,180))
    pic_image.save(picture_path)
    return picture_to_save

@app.route("/account", methods=['GET','POST'])
@login_required
def account():
    # if current_user.is_authenticated:
    form=UpdateAccountForm()
    if form.validate_on_submit():
        current_user.username=form.username.data
        current_user.email=form.email.data
        if form.picture.data:
            pic=saveProfile(form.picture.data)
            old_pic=current_user.image_file
            current_user.image_file=pic
            if old_pic!="default.jpg":
                old_path=os.path.join(app.root_path, "static/profiles/",old_pic)
                os.remove(old_path)
    
        db.session.commit()
        flash("Account updated successfully.",'success' )
        return redirect(url_for("account")) #post get redirect pattern
    elif request.method=="GET":
        form.username.data=current_user.username
        form.email.data=current_user.email
    profile=url_for("static", filename="profiles/"+current_user.image_file)
    return render_template("account.html", form=form, title="Pepper: Account",profile=profile)
    # else:
    #     return redirect(url_for('login'))


@app.route("/newpost", methods=["GET", "POST"])
@login_required
def newpost():
    form=NewPostForm()
    if form.validate_on_submit():
        flash("Post sent.")
        return redirect(url_for('home'))
    return render_template("newpost.html", form=form, title="Pepper: Creat New Post")