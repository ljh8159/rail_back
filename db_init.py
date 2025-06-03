import sqlite3

conn = sqlite3.connect('reports.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lat REAL,
    lng REAL,
    address TEXT,
    timestamp TEXT,
    stage INTEGER      -- 단계(1~4)
)
''')

conn.commit()
conn.close()
print("✅ DB 초기화 완료")
