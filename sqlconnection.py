from sqlalchemy import create_engine
import psycopg2
import io


def saveframe_postgresql(df):
    engine = create_engine('postgresql+psycopg2://username:password@host:port/database')
    df.head(0).to_sql('table_name', engine, if_exists='replace', index=False)  # truncates the table

    conn = engine.raw_connection()
    cur = conn.cursor()
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    contents = output.getvalue()
    cur.copy_from(output, 'table_name', null="")  # null values become ''
    conn.commit()