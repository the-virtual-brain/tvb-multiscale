# -*- coding: utf-8 -*-
import h5py


class Base(object):

    h5_path = ""

    @property
    def _hdf_file(self):
        return None

    def _set_hdf_file(self, hfile):
        pass

    @property
    def _fmode(self):
        return None

    @property
    def _mode(self):
        return ""

    @property
    def _mode_past(self):
        return ""

    @property
    def _to_from(self):
        return ""

    def _open_file(self, type_name=""):
        try:
            self._set_hdf_file(h5py.File(self.h5_path, self._fmode, libver='latest'))
        except Exception as e:
            self.logger.warning("Could not open file %s\n%s!" % (self.h5_path, str(e)))

    def _assert_file(self, path=None, type_name=""):
        if path is not None:
            self.h5_path = path
        if self._hdf_file is not None:
            self._close_file()
        self._open_file(type_name)
        self.logger.info("Starting to %s %s %s H5 file: %s" % (self._mode, type_name, self._to_from, self.h5_path))

    def _close_file(self, close_file=True):
        if close_file:
            try:
                self._hdf_file.close()
                self._set_hdf_file(None)
            except Exception as e:
                self.logger.warning("Could not close file %s\n%s!" % (self.h5_path, str(e)))

    def _log_success_or_warn(self, exception=None, type_name=""):
        if exception is None:
            self.logger.info("Successfully %s %s %s H5 file: %s" %
                             (self._mode_past, type_name, self._to_from, self.h5_path))
        else:
            self.logger.warning("Failed to %s %s %s H5 file %s\n%s!" %
                                (self._mode, type_name, self._to_from, self.h5_path, str(exception)))
