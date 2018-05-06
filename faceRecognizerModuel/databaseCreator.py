import sqlite3


con = sqlite3.connect("faceDatabase.db")

c=con.cursor()


sql=""" 
DROP TABLE IF EXISTS knownPeople;
CREATE TABLE knownPeople( ID INTEGER UNIQUE PRIMARY KEY AUTOINCREMENT, NAME TEXT);
"""

c.executescript(sql)

con.commit()

con.close()
