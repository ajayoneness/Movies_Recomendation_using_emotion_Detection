from django.urls import path
from . import views

urlpatterns = [
    path('',views.home ,name="home"),
    path('cam/',views.opencamera ,name="opencamera"),
    path('movieslist/<str:emotion>',views.moviesList ,name="moviesList"),
]
