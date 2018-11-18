from sqlalchemy.ext.automap import automap_base

Base = automap_base()

from sqlalchemy import create_engine

engine = create_engine('sqlite:///Chinook_Sqlite.sqlite')

Base.prepare(engine, reflect=True)

Base.classes.keys()

Artist = Base.classes.Artist
Album = Base.classes.Album

from sqlalchemy.orm import Session

session = Session(engine)
for artist in session.query(Artist).limit(10):
    print(artist.ArtistId, artist.Name)

artist = session.query(Artist).first()
for album in artist.album_collection:
    print('{} - {}'.format(artist.Name, album.Title))