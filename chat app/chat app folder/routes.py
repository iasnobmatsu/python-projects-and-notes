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


@app.route("/",methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    page=request.args.get("page", 1, type=int)
    posts=Post.query.order_by(Post.date_posted.desc()).paginate(page=page,per_page=5)
    return render_template('home.html', posts=posts, title="Pepper: Blog, Share, More")



@app.route("/explore")
def explore():
    posts=Post.query.all()
    top_posts=[]
    for post in posts:
        likes_query=Like.query.filter_by(postid=post.id)
        likes_num=0
        if likes_query.first():
            for i in likes_query:
                likes_num+=1
                top_posts.append((post,likes_num))
        else:
            pass
    if len(top_posts)>50:
        top_posts=top_posts[:50]
    top_posts=sorted(top_posts, reverse=True, key=lambda post: post[1] )
    return render_template('explore.html', title="Pepper:Explore",top_posts=top_posts,posts=posts)


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
        ismarkdown=0
        if form.ismarkdown.data:
            ismarkdown=1
        else:
            ismarkdown=0
        post=Post(title=form.title.data, ismarkdown=ismarkdown, content=form.article.data,author=current_user)
        db.session.add(post)
        db.session.commit()
        flash("Post sent.","success")
        return redirect(url_for('home'))
    return render_template("newpost.html", legend="New Post", form=form, title="Pepper: Creat New Post")

@app.route("/post/<int:post_id>",methods=["GET","POST"])
def post(post_id):
    likeform=LikeForm()
    post=Post.query.get_or_404(post_id)
    liked=False
    if current_user.is_authenticated:
        like=Like.query.filter_by(userid=current_user.id, postid=post_id).first()
        if like:
            liked=True

    if request.method == 'POST':
        if not current_user.is_authenticated:
            flash("Login to like a post.","warning")
            return redirect(url_for('login'))
        userid=current_user.id
        postid=post_id
            
        if Like.query.filter_by(userid=userid, postid=postid).first():
            db.session.delete(Like.query.filter_by(userid=userid, postid=postid).first())
            db.session.commit()
            liked=False
            
        else:
            like=Like(userid=userid,postid=postid)
            db.session.add(like)
            db.session.commit()
            liked=True
    
    if post.author==current_user:
        return render_template("post.html",liked=liked,likeform=likeform, title=post.title, post=post,edit="Edit",delete="Delete")
    return render_template("post.html", liked=liked,likeform=likeform, title=post.title, post=post,edit="", delete="")


@app.route("/post/<int:post_id>/edit",methods=["GET", "POST"])
@login_required
def edit(post_id):
    post=Post.query.get_or_404(post_id)
    if post.author!=current_user:
        abort(403)
    form=NewPostForm()
    if form.validate_on_submit():
        ismarkdown=0
        if form.ismarkdown.data:
            ismarkdown=1
        else:
            ismarkdown=0
        post.ismarkdown=ismarkdown
        post.title=form.title.data
        post.content=form.article.data
        db.session.commit()
        flash("Updated", "success")
        return redirect(url_for('post',post_id=post.id))
    elif request.method=="GET":
        form.title.data=post.title
        form.article.data=post.content
        form.ismarkdown.data=post.ismarkdown
    return render_template("newpost.html", legend="Edit Post", form=form, title="Pepper: Edit Post")
    
@app.route("/post/<int:post_id>/delete",methods=["GET", "POST"])
@login_required
def delete(post_id):
    post=Post.query.get_or_404(post_id)
    if post.author!=current_user:
        abort(403)
    db.session.delete(post)
    likes=Like.query.filter_by(postid=post_id)
    for like in likes:
        db.session.delete(like)
    db.session.commit()
    flash("Post deleted","success")
    return redirect(url_for("home"))


@app.route("/user/<string:username>")
def userpage(username):
    page=request.args.get("page", 1, type=int)
    user=User.query.filter_by(username=username).first_or_404()
    posts=Post.query.filter_by(author=user).order_by(Post.date_posted.desc()).paginate(page=page,per_page=5)
    return render_template("userpage.html",posts=posts, title=username,username=username)

@app.route("/mypage")
@login_required
def mypage():
    page=request.args.get("page", 1, type=int)
    user=current_user
    posts=Post.query.filter_by(author=user).order_by(Post.date_posted.desc()).paginate(page=page,per_page=5)
    return render_template("mypage.html",posts=posts, title=user.username,username=user.username)



@app.route("/linkforreset", methods=['GET','POST'])
def resetemail():
    form=ResetEmailForm()
    if form.validate_on_submit():
        user=User.query.filter_by(email=form.email.data).first()
        flash("Reset email sent." 'success')
        return redirect(url_for('login'))
    return render_template("resetemail.html", title="Password Reset", form=form)

@app.route("/resetpassword/<token>", methods=['GET','POST'])
def resetpassword(token):
    user=User.verify_token(token)
    if user is None:
        flash('invalid or expired token',"warning")
        return redirect(url_for('resetemail'))
    form=ResetPasswordForm()
    return render_template("resetpassword.html", title="Password Reset", form=form)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.route("/likes")
@login_required
def likes():
    username=current_user.username
    
    posts=[]
    likes=Like.query.filter_by(userid=current_user.id)
    for like in likes:
        post=Post.query.get(like.postid)
        posts.append(post)
    
    return render_template("likes.html",posts=posts, title="Liked posts by "+ username,username=username)


@app.route("/mkdpost")
@login_required
def mkdpost():
    return render_template("mkdpost.html", title="create a new markdown page")