# ============================
# IMPORTING REQUIRED LIBRARIES
# ============================

import os
import io
import re
import base64
import pickle
import mimetypes
from io import BytesIO

from PIL import Image
from tqdm import tqdm
from flask import (
    Flask, render_template, request, url_for, send_file,
    Response, session, abort
)
from bson import ObjectId, Binary
from bson.errors import InvalidId
from pinecone import Pinecone, ServerlessSpec

# ============================
# LOCAL IMPORTS
# ============================

from db import fs, user, user_images_collection
from utils import extract_features, prs_recommender, recommender, pc

# ============================
# FLASK APP INITIALIZATION
# ============================

app = Flask(__name__)
app.secret_key = "jigysa"


# ============================
# AUTHENTICATION ROUTES
# ============================

@app.route('/')
def login():
    return render_template('login.html')


@app.route('/cred', methods=['POST'])
def cred():
    Email = request.form.get('Email')
    Password = request.form.get("Password")

    credentials = {'Email': Email, 'Password': Password}
    logged_user = user.find_one(credentials)

    if logged_user:
        session['user_id'] = str(logged_user['_id'])
        session['username'] = logged_user.get('Name', 'default_user')
        return render_template('home.html', im=url_for('static', filename="home_image.png"))
    else:
        return render_template('login.html', r1="User not Registered")


# ============================
# REGISTRATION ROUTES
# ============================

@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/r', methods=['POST'])
def reg():
    data = {
        'Name': request.form.get('Name'),
        'Contact': request.form.get("Contact"),
        'Email': request.form.get('Email'),
        'Password': request.form.get("Password")
    }
    user.insert(data)
    return render_template('register.html', r0="Your Registration is Done")


# ============================
# HOME ROUTES
# ============================

@app.route('/home')
def home():
    return render_template('home.html')


# ============================
# IMAGE SERVING ROUTES
# ============================

@app.route('/image/<image_id>')
def get_image(image_id):
    try:
        obj_id = ObjectId(image_id)
        image_doc = user_images_collection.find_one({'_id': obj_id})
        if not image_doc or 'image' not in image_doc:
            return abort(404)

        return send_file(io.BytesIO(image_doc['image']), mimetype='image/jpeg')
    except Exception as e:
        print(f"Image fetch error: {e}")
        return abort(500)


@app.route("/file/image/<file_id>")
def get_image_fileid(file_id):
    try:
        file_obj = fs.get(ObjectId(file_id))
        return Response(file_obj.read(), content_type=file_obj.content_type)
    except Exception as e:
        return {"error": str(e)}, 404


# ============================
# PERSONALIZED RECOMMENDER SYSTEM (PRS)
# ============================

@app.route('/prs_submit', methods=['POST'])
def prs_submit():
    username = session.get('username', 'default_user')
    images = request.files.getlist('files')

    if not images or all(f.filename == '' for f in images):
        return "No selected files", 400

    prs_index = pc.Index("virtualwardrobe")

    for f in tqdm(images):
        if f and f.filename:
            file_bytes = f.read()
            file_obj = BytesIO(file_bytes)

            feature = extract_features(file_obj)

            prs_index.upsert(
                vectors=[{
                    "id": f"{username}/{f.filename}",
                    "values": feature.tolist(),
                    "metadata": {
                        "username": username,
                        "filename": f.filename
                    }
                }],
                namespace=f"{username}_wardrobe"
            )

            user_images_collection.insert_one({
                "username": username,
                "filename": f.filename,
                "image": Binary(file_bytes)
            })

    return prs_fashion_recommend()


@app.route('/prs_result', methods=['POST'])
def prs_result():
    image = request.files.get('UploadImage')
    if image is None or image.filename == '':
        return "No file selected", 400

    feature = extract_features(image.stream)
    df = prs_recommender(feature, session['username'])

    usernames = df['username'][:5].tolist()
    filenames = df['filename'][:5].tolist()

    image_urls = []
    for username, filename in zip(usernames, filenames):
        doc = user_images_collection.find_one({
            "username": username,
            "filename": filename
        })
        image_urls.append(url_for('get_image', image_id=str(doc['_id'])) if doc else None)

    username = session.get('username', 'default_user')
    user_images = user_images_collection.find({"username": username})
    wardrobe_images = [url_for('get_image', image_id=str(img['_id'])) for img in user_images]

    return render_template("PRS_Fashion_Recommender_Result.html",
                           s0=image_urls[0], s1=image_urls[1], s2=image_urls[2],
                           s3=image_urls[3], s4=image_urls[4],
                           wardrobe_images=wardrobe_images)


@app.route('/home/prs_fashion_recommend')
def prs_fashion_recommend():
    username = session.get('username', 'default_user')
    user_images = user_images_collection.find({"username": username})
    wardrobe_images = [url_for('get_image', image_id=str(img['_id'])) for img in user_images]

    return render_template('PRS_Fashion_Recommender.html', wardrobe_images=wardrobe_images)


# ============================
# GENERAL FASHION RECOMMENDER
# ============================

@app.route('/home/fashion_recommend')
def fr():
    return render_template('Fashion_Recommender.html')


@app.route('/home/fashion_recommend/result', methods=['POST'])
def result():
    f = request.files['Image']

    file_id = fs.put(f, filename=f.filename, content_type=f.content_type)
    image_data = fs.get(file_id).read()

    file_stream = BytesIO(image_data)
    features = extract_features(file_stream)
    df = recommender(features)

    IMG_0, IMG_1, IMG_2, IMG_3, IMG_4 = df['Product_Image'][:5]
    L0, L1, L2, L3, L4 = df['Product_Link'][:5]
    upload_image_url = f"/image/{file_id}"

    print(upload_image_url)

    return render_template(
        'Fashion_Recommender.html',
        image_url=upload_image_url,
        s0=IMG_0, s1=IMG_1, s2=IMG_2, s3=IMG_3, s4=IMG_4,
        h0=L0, h1=L1, h2=L2, h3=L3, h4=L4,
    )


# ============================
# APP ENTRY POINT
# ============================

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=3000)
