#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include"../dependencies/fast-cpp-csv-parser/csv.h"
#include <iostream>
#include <iterator>
#include <vector>
#include "../src/LatentFactorModel.h"

template<typename T>
void passValuesToLTF (std::vector<int> &users, std::vector<int> &items, std::vector<T> &ratings, int user, int item, float &result) {
  model::LatentFactorModel<T>* m = new model::LatentFactorModel<T>(users, items, ratings);
  m = m->build(5);
  int i = 0;
  while (i < 5) {
    m = m->iterate(0.01, 0.05);
    ++i;
  }
  result = m->predict(user, item);
}

static PyObject * predict_wrapper(PyObject * self, PyObject * args)
{
  Py_buffer users;
  Py_buffer items;
  Py_buffer ratings;
  int user;
  int item;

  float result;
  char * strresult;
  PyObject * ret;

  // parse arguments
  if (!PyArg_ParseTuple(args, "y*y*y*ii", &users, &items, &ratings, &user, &item)) {
    return NULL;
  }

  std::vector<int> usersVector(users.len);
  std::vector<int> itemsVector(items.len);
  std::vector<float> ratingsVector(ratings.len);

  int i = 0;
  for (i = 0; i < users.len / users.itemsize; i++) {
      usersVector.push_back((static_cast<int*>(users.buf))[i]);
      itemsVector.push_back((static_cast<int*>(items.buf))[i]);
      ratingsVector.push_back((static_cast<float*>(ratings.buf))[i]);
  }

  passValuesToLTF<float>(usersVector, itemsVector, ratingsVector, user, item, result);
  
  asprintf(&strresult, "%g", result);

  // build the resulting string into a Python object.*/
  ret = PyUnicode_FromString(strresult);

  return ret;
}

static PyMethodDef PcfMethods[] = {
    {"predict", predict_wrapper, METH_VARARGS, "Predict 1 rating."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef pcfmodule = {
    PyModuleDef_HEAD_INIT,
    "pcf",   /* name of module */
     NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    PcfMethods
};

PyMODINIT_FUNC
PyInit_pcf(void)
{
    return PyModule_Create(&pcfmodule);
}