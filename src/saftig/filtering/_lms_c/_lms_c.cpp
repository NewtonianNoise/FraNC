// Relevant documentation: https://docs.python.org/3/extending/newtypes_tutorial.html

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <vector>
#include <iostream>
#include <cmath>

typedef struct {
    PyObject_HEAD;
    unsigned int n_filter, idx_target, n_channel;
    double step_scale, clip_coefficients;
    bool normalized;

    std::vector<std::vector<double>> filter_coefficients;
    /* Type-specific fields go here. */
} LMS_C_OBJECT;

/**
 * check that the array is 2D with the given shape and dtype double
 * raise an exception if not
 * returns false if an exception was raised
 */
bool check_array_properties(PyArrayObject *array, int shape0, int shape1) {
    // check dtype
    if(PyArray_TYPE(array) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "np.float64 is the only supported dtype");
        return false;
    }

    // Check that it's a 2D array
    if (PyArray_NDIM(array) != 2) {
        std::cout << "2+" << std::endl;
        PyErr_SetString(PyExc_ValueError, "Input must be a 2D array.");
        return false;
    }

    // check that dimensions match
    npy_intp* dims = PyArray_DIMS(array);
    int channels = dims[0], n_filter = dims[1];

    if(channels != shape0) {
        PyErr_SetString(PyExc_ValueError, "Input channel count missmatch");
        return false;
    }
    if(n_filter != shape1) {
        PyErr_SetString(PyExc_ValueError, "Input sample count missmatch");
        return false;
    }
    return true;
}

static PyObject *
LMS_C_step(LMS_C_OBJECT *self, PyObject *args)
{
    PyArrayObject* array;
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    NpyIter_GetMultiIndexFunc *get_multi_index;
    npy_intp multi_index[2];
    double** dataptr;
    double target;

    // get parameter
    if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &array, &target)) {
        return NULL;
    }

    if(not check_array_properties(array, self->n_channel, self->n_filter)) {
        return NULL;
    }

    // get iterator
    iter = NpyIter_New( array,
            NPY_ITER_READONLY | NPY_ITER_MULTI_INDEX | NPY_ITER_REFS_OK,
            NPY_KEEPORDER,
            NPY_NO_CASTING,
            NULL);
    if (iter == NULL) {
        return NULL;
    }

    if (NpyIter_GetIterSize(iter) == 0) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    dataptr = (double**)NpyIter_GetDataPtrArray(iter);
    get_multi_index = NpyIter_GetGetMultiIndex(iter, NULL);

    if ((iternext == NULL) || (dataptr == NULL) || (get_multi_index == NULL)) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    // calculate prediction
    double prediction = 0, normalization = 0;
    if(self->normalized) {
        do {
            get_multi_index(iter, multi_index);

            // the following uses the fma() instruction that can be faster on some computers
            // prediction += (**dataptr) * self->filter_coefficients[multi_index[0]][multi_index[1]];
            prediction = fma((**dataptr), self->filter_coefficients[multi_index[0]][multi_index[1]], prediction);
            // normalization += (**dataptr) * (**dataptr);
            normalization = fma(**dataptr, **dataptr, normalization);
        } while (iternext(iter));
    } else {
        do {
            get_multi_index(iter, multi_index);
            prediction += (**dataptr) * self->filter_coefficients[multi_index[0]][multi_index[1]];
        } while (iternext(iter));
        normalization = 1;
    }

    // calculate instantaneous prediction error
    double error = target - prediction;

    // reset iterator to the start of the numpy array
    char reset_fail_msg[] = "Iterator reset failed";
    if(NpyIter_Reset(iter, (char**)&reset_fail_msg) != NPY_SUCCEED) {
        return NULL;
    }

    // update filter
    do {
        get_multi_index(iter, multi_index);

        self->filter_coefficients[multi_index[0]][multi_index[1]] += 2 * self->step_scale * error * (**dataptr) / normalization;

        // clip the filter coefficients to self->clip_coefficients if the value is not NaN
        if(!std::isnan(self->clip_coefficients)) {
            if(self->filter_coefficients[multi_index[0]][multi_index[1]] > self->clip_coefficients) {
                self->filter_coefficients[multi_index[0]][multi_index[1]] = self->clip_coefficients;
            } else if(self->filter_coefficients[multi_index[0]][multi_index[1]] < -self->clip_coefficients) {
                self->filter_coefficients[multi_index[0]][multi_index[1]] = -self->clip_coefficients;
            }
        }
    } while (iternext(iter));

    // dealloc the iter instance
    if (!NpyIter_Deallocate(iter)) {
        return NULL;
    }

    return PyFloat_FromDouble(prediction);
}

static PyMethodDef LMS_C_methods[] = {
    {"step",
     (PyCFunction) LMS_C_step,
     METH_VARARGS,
     "Return the name, combining the first and last name", },
    {NULL}  /* Sentinel */
};

static int
LMS_C_init(LMS_C_OBJECT *self, PyObject *args, PyObject *kwds)
{
    static char * kwlist[] = { (char *) "n_filter",
                               (char *) "idx_target",
                               (char *) "n_channel",
                               (char *) "step_scale",
                               (char *) "normalized",
                               (char *) "coefficient_clipping",
                               (char *) NULL}; // must be terminated with a NULL

    self->normalized = true;
    self->clip_coefficients = std::nan("");

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "IIId|pd", kwlist,
                                     &self->n_filter,
                                     &self->idx_target,
                                     &self->n_channel,
                                     &self->step_scale,
                                     &self->normalized,
                                     &self->clip_coefficients)) {
        return -1;
    }

    // set the filter size and reset all coefficients to zero
    self->filter_coefficients.insert(self->filter_coefficients.begin(),
            self->n_channel,
            std::vector<double>(self->n_filter, 0));

    return 0;
}

static void
LMS_C_dealloc(LMS_C_OBJECT *self)
{
    // this must call Py_XDECREF(object) for all held python objects
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyTypeObject LMS_C_TYPE = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "LMS_C.LMS_C",
    .tp_basicsize = sizeof(LMS_C_OBJECT),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) LMS_C_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyDoc_STR("LMS Filter implemented in C"),
    .tp_methods = LMS_C_methods,
    .tp_init = (initproc) LMS_C_init,
    .tp_new = PyType_GenericNew,
};

static int
custom_module_exec(PyObject *m)
{
    if (PyType_Ready(&LMS_C_TYPE) < 0) {
        return -1;
    }

    if (PyModule_AddObjectRef(m, "LMS_C", (PyObject *) &LMS_C_TYPE) < 0) {
        return -1;
    }

    return 0;
}

static PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, (void *) custom_module_exec},
    // Just use this while using static types
#if PY_VERSION_HEX >= 0x030C0000
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
    {0, NULL},
};

static struct PyModuleDef _lms_c = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "lms_c",
    .m_doc = "An LMS filter module",
    .m_size = 0,
    .m_slots = module_slots,
};

PyMODINIT_FUNC PyInit__lms_c(void)
{
    import_array(); // enable numpy support, otherwise we get a lot of segfaults
    return PyModuleDef_Init(&_lms_c);
}
