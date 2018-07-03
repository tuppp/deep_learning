import FunctionLibraryExtended as library
tmp=library.getConnectionDWD()
print(library.getMaxTempUp(30.0,tmp[0],tmp[1]))
#Anfrage PLZ
result=tmp[1].execute("""SELECT station_id,postcode, station_name, count(*)
                        FROM dwd
                         GROUP BY station_name
                         """)
#result=tmp[1].execute("""SELECT station_id,postcode, station_name,measure_date
#                        FROM dwd
#                         Where station_name='Berlin-Tegel'
#                         ORDER BY measure_date ASC
#                         """)
for line in result:
    print(line)

#result=tmp[1].execute("""SELECT station_name,measure_date,average_temp
#                        FROM dwd
#                         Where station_name='' OR station_name='' OR station_name='' OR station_name='' OR station_name
#                         ORDER BY measure_date ASC, station_name ASC
#                         """)
#