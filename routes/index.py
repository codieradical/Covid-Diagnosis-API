import codecs
from flask import request, abort, redirect, render_template, send_from_directory, Markup
import os

def register_routes(app):
    global static_folder

    # These are imported after serve is defined.
    from routes.model import register_routes as register_model_routes

    register_model_routes(app)