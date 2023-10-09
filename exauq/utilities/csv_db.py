from __future__ import annotations

import csv
import pathlib
from collections.abc import Callable, Collection, Iterator
from os import PathLike
from typing import Any, Optional, TypeAlias, Union

Record: TypeAlias = dict[str, str]
Path: TypeAlias = Union[str, PathLike]


class CsvDB(object):
    """A class for interfacing with a simple database backed by a csv file.

    A `CsvDB` instance is initialised with a file path that defines a csv file, either
    an existing file or one that doesn't yet exist. In the case where the file exists, the
    file will be used as the underlying database file where records are recorded and
    retrieved. In this case, it is assumed that the first row of the csv file contains the
    field names of the database and these are checked against the fields supplied during
    initialisation.

    If the csv file doesn't exist at the supplied path, then it will be created when
    adding the first record to the database via the `create()` method.

    Parameters
    ----------
    path : str or os.PathLike
        The file path to a csv file.
    fields : Collection[str]
        A collection of field names to be used for the database. This should not contain
        repeated field names or the empty string. If the csv file at `path` exists, then
        `fields` should match the fields defined as the header row in the file.

    Raises
    ------
    ValueError
        If the supplied `fields` is empty, contains empty strings or contains repeated
        strings.
    MissingFieldsError
        If the csv database file at `path` contains empty field names.
    RepeatedFieldsError
        If the csv database file at `path` contains repeated field names in its first row.
    FieldsMismatchError
        If the supplied `fields` does not define the same set of field names as present
        in the csv file at `path`.
    """

    def __init__(self, path: Path, fields: Collection[str]):
        self._path = pathlib.Path(path)
        self._fields_arg = "fields"  # should match the name of the argument in __init__
        self._fields = self._make_fields(fields)

    def _make_fields(self, fields: Collection[str]) -> None:
        """Make the fields to store reference in database operations.

        Along the way, this method performs checks on the supplied fields and, if present,
        those in the csv file. The checks performed are:

        * The collection of field names is not empty.
        * None of the field names are missing, i.e. the empty string.
        * None of the field names are repeated.
        * The supplied fields agree with those in the csv file, if the latter is present.

        If all relevant checks pass, then the fields as ordered in the csv file are
        returned. Otherwise, the supplied `fields` are returned.
        """

        if len(fields) == 0:
            raise ValueError(
                f"Argument '{self._fields_arg}' defines an empty collection."
            )

        if self._any_missing(fields):
            raise ValueError(f"Argument '{self._fields_arg}' contains empty field names.")

        if repeated_fields := self._make_repetitions_str(fields):
            raise ValueError(
                f"Argument '{self._fields_arg}' contains repeated fields: {repeated_fields}."
            )

        if not self._path.exists():
            return fields

        with open(self._path, mode="r") as csvfile:
            file_fields = tuple(csvfile.readline().strip().split(","))

        if self._any_missing(file_fields):
            raise MissingFieldsError(
                f"Database file {self._path} contains empty field names."
            )

        if repeated_fields := self._make_repetitions_str(file_fields):
            raise RepeatedFieldsError(
                f"Database file {self._path} contains repeated fields: {repeated_fields}."
            )

        if not set(fields) == set(file_fields):
            raise FieldsMismatchError(
                f"'{self._fields_arg}' does not agree with the fields defined in {self._path}."
            )

        return file_fields

    @staticmethod
    def _any_missing(coll: Collection[str]) -> bool:
        """Returns ``True`` if the empty string is in the collection `coll`."""

        return any(x == "" for x in coll)

    @staticmethod
    def _make_repetitions_str(coll: Collection[str]) -> str:
        """Make a string representation of repeated elements, that is suitable for use in
        error messages."""

        return ", ".join(sorted({f"'{x}'" for x in coll if coll.count(x) > 1}))

    def create(self, record: dict[str, Any]) -> None:
        """Add a new record to the underlying database csv file.

        If the underlying csv file does not exist then it will be created before adding
        the record to it. Values are converted to strings by using the ``str()`` function
        before writing to the database. In particular, ``None`` values are converted to
        the empty string.

        Parameters
        ----------
        record : dict[str, Any]
            The record data to write to the database. The keys should consist of the
            field names in the database and the values should be the values to record
            against the corresponding field.

        Raises
        ------
        ValueError
            If the record is missing any of the fields of the database.
        """

        missing_fields = ", ".join([f"'{k}'" for k in self._fields if k not in record])
        if missing_fields:
            raise ValueError(
                f"Argument 'record' missing the following fields: {missing_fields}."
            )
        mode = "a" if self._path.exists() else "x"
        with open(self._path, mode=mode, newline="") as csvfile:
            writer = csv.DictWriter(csvfile, self._fields)
            if mode == "x":
                writer.writeheader()
            writer.writerow(record)

    def retrieve(self, field: str, value: Any) -> Optional[Record]:
        """Retrieve the first record with a specified value for a field.

        If no record such record is found, then ``None`` is returned. Database records are
        ordered according to the order in which they were added, in a 'first in, first
        out' fashion.

        Parameters
        ----------
        field : str
            The field against which to look up the given `value`.
        value : Any
            The value to match on. This will be coverted to a string before performing
            the lookup.

        Returns
        -------
        dict[str, str] or None
            The first record whose value at the given `field` matches `value` (after
            string conversion, if necessary), or ``None`` if no such record was found.

        Raises
        ------
        DatabaseLookupError
            If the field supplied does not exist in the database.

        Notes
        -----
        The main use case for `retrieve()` is to retrieve records by field values that
        uniquely define records. In other cases, it is possible that `retrieve()` will not
        return the record desired, since it just returns the first matching record it
        finds with `field` set to `value`.
        """

        self._validate_field(field)
        if not self._path.exists():
            return None

        with open(self._path, mode="r", newline="") as csvfile:
            reader = self._make_data_reader(csvfile)
            for row in reader:
                if row[field] == str(value):
                    return row

    def _validate_field(self, field: str) -> None:
        """Check that the given field is present in the database."""

        if field not in self._fields:
            raise DatabaseLookupError(
                f"'{field}' does not define a field in the database."
            )

    def _make_data_reader(self, csvfile: Iterator[Record]) -> csv.DictReader:
        """Make a reader that iterates through all non-header records."""

        reader = csv.DictReader(csvfile, self._fields)
        _ = next(reader)
        return reader

    def query(
        self, predicate_fn: Optional[Callable[[Record], bool]] = None
    ) -> tuple[Record]:
        """Return all records matching some condition.

        The condition to filter the records on should be expressed as a function that maps
        records to boolean values. This will be evaluated on each record in the database
        and only those records evaluating to ``True`` will be returned.

        Parameters
        ----------
        predicte_fn : Callable[[dict[str, str]], bool], optional
            (Default: ``None``) A function that maps a record to a boolean value. If
            ``None`` then all records in the database will be returned.

        Returns
        -------
        tuple[dict[str, str]]
            All records in the database on which the `predicate_fn` evaluates to ``True``.

        Raises
        ------
        DatabaseLookupError
            If the predicate function attempts to access a field that is not present in
            the database.
        """

        if not self._path.exists():
            return tuple()
        with open(self._path, mode="r", newline="") as csvfile:
            try:
                return tuple(filter(predicate_fn, self._make_data_reader(csvfile)))
            except KeyError as exc:
                raise DatabaseLookupError(
                    "Bad 'predicate_fn': attempted to look up a field not in the database."
                ) from exc
            except Exception as exc:
                raise exc.__class__(f"Bad 'predicate_fn': {exc}") from exc

    def update(self, field: str, value: Any, record: dict[str, Any]) -> None:
        """Update a record matching a given field/value combination.

        The first record in the database whose `field` entry matches `value` (after string
        conversion) will be replaced with the given record. Database records are ordered
        according to the order in which they were added, in a 'first in, first out'
        fashion. As with the `create()` method, values in `record` are converted to
        strings by using the ``str()`` function before writing to the database; in
        particular, ``None`` values are converted to the empty string.

        Parameters
        ----------
        field : str
            The field against which to look up the given `value` when locating the
            record to replace.
        value : Any
            The value to match on when looking up the record to replace. This will be
            coverted to a string before performing the lookup.
        record : dict[str, Any]
            The new record with which to replace the old record. The keys should consist
            of the field names in the database and the values should be the values to
            record against the corresponding field.

        Raises
        ------
        DatabaseLookupError
            If one of the following occurs:
            * The field supplied does not exist in the database.
            * A record cannot be found with the given `field`/`value` combination.

        Notes
        -----
        As with the `retrieve()` method, the main use case of `update()` is to update
        records by field values that uniquely define records. In other cases, it is
        possible that `update()` will not update the desired record, since it just updates
        the first matching record it finds with `field` set to `value`.
        """

        self._validate_field(field)
        records = list(self.query())
        field_values = tuple(rec[field] for rec in records)
        try:
            records[field_values.index(str(value))] = record
        except ValueError as exc:
            raise DatabaseLookupError(
                f"Could not find record with {field} = {value}."
            ) from exc

        with open(self._path, mode="w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, self._fields)
            writer.writeheader()
            writer.writerows(records)


class FieldsMismatchError(Exception):
    """Raised when two collections of database field names do not agree."""

    pass


class RepeatedFieldsError(Exception):
    """Raised when a collection of field names contains repetitions."""

    pass


class MissingFieldsError(Exception):
    """Raised when a collection of field names is empty."""

    pass


class DatabaseLookupError(Exception):
    """Raised when attempting to look up a non-existant record in a database."""

    pass
