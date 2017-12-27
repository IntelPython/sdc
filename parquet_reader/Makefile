hpat_parquet_reader: hpat_parquet_reader.o
	${CXX} ${LDFLAGS} -shared -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 hpat_parquet_reader.o -L${PREFIX}/lib -lparquet -larrow -o ${PREFIX}/lib/libhpat_parquet_reader.so

hpat_parquet_reader.o: hpat_parquet_reader.cpp
	${CXX} ${CPPFLAGS} -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -I${PREFIX}/include -c hpat_parquet_reader.cpp -o hpat_parquet_reader.o

clean:
	rm -f *.o *~ ${PREFIX}/lib/libhpat_parquet_reader.so
