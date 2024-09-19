# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Utilities for working with threads.


.. autosummary::

    StoppableThread
    ThreadedHTTPServer

"""

from http.server import BaseHTTPRequestHandler, HTTPServer
from socket import socket
from threading import Event, Thread
from typing import Callable, Optional, Tuple, Type, TypeVar

__all__ = ["StoppableThread", "ThreadedHTTPServer"]


ST = TypeVar("ST", bound="StoppableThread")


class StoppableThread(Thread):
    """:class:`~threading.Thread` with some more stopping-sugar.

    Using this class makes the most sense when inheriting from it instead of
    passing a ``target`` keyword argument to the constructor, as one has to
    actively make use of :meth:`stopped` within :meth:`~threading.Thread.run`
    in order for :meth:`stop` to work as intended.

    Besides the :meth:`stop` method, this also supports `context manager style
    <https://docs.python.org/3.6/reference/compound_stmts.html#with>`_ usage
    calling :meth:`stop` and :meth:`~threading.Thread.join` on context exit::

        >>> import time
        >>> class Waiter(StoppableThread):
        ...     def run(self):
        ...         self.i = 0
        ...         while not self.stopped():
        ...             self.i += 1
        >>>
        >>> waiter = Waiter()
        >>> waiter.start()
        >>> waiter.stop().join()
        >>>
        >>> with Waiter() as waiter:
        ...     time.sleep(0.01)
        ...     assert waiter.i > 0

    All constructor arguments are keyword arguments only, as some of the
    original :class:`threading.Thread` arguments and their order can be quite
    confusing.

    Args:
        name: Optional name for the thread, helpful for debugging.
        daemon: Whether to run the thread as a daemon, meaning that it will be
            forcefully shutdown when the main program is quit.
        target: An optional method to use in-place of the :meth:`run` member
            method. Helpful for slim threads where one would not want to
            actually implement a class inheriting from this one.

    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        daemon: Optional[bool] = None,
        target: Optional[Callable] = None
    ):
        super().__init__(name=name, daemon=daemon, target=target)
        self._stop_event = Event()

    def __enter__(self: ST) -> ST:
        """Starts the thread when entering a context."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Joins the thread when leaving a context."""
        self.stop().join()

    def __del__(self):
        """Stops the thread on deletion."""
        self.stop()

    def stop(self: ST) -> ST:
        """Sends the stop event to the running :meth:`~threading.Thread.run` method.

        Can be chained with :meth:`~threading.Thread.join`::

            thread = StoppableThreadSubclass()
            thread.start()
            thread.stop().join()

        """
        self._stop_event.set()
        return self

    def stopped(self) -> bool:
        """Checks whether the thread has been stopped.

        Should be used in :meth:`~threading.Thread.run`.
        """
        return self._stop_event.is_set()


THS = TypeVar("THS", bound="ThreadedHTTPServer")


class ThreadedHTTPServer(HTTPServer):
    """A :class:`~http.server.HTTPServer` serving from its own thread.

    All requests are handled synchronously one after another -- this does not
    handle requests in individual threads or processes. All it does is not block
    the main (initializing) thread.

    This could be extended to also handle requests in their own threads by
    inheriting from :class:`~http.server.ThreadingHTTPServer`, but that one
    apparently entails some risks regarding scalability as it creates a new
    thread per request instead of employing a fixed pool of threads to work
    off the incoming requests one by one.

    Example::

        >>> from http.server import BaseHTTPRequestHandler
        >>> from urllib import request
        >>>
        >>> class RequestHandler(BaseHTTPRequestHandler):
        ...     def do_GET(self) -> None:
        ...         self.send_response(200)
        ...         self.end_headers()
        ...         self.wfile.write("response".encode())
        >>>
        >>> httpd = ThreadedHTTPServer(("localhost", 12364), RequestHandler).start()
        >>> request.urlopen("http://localhost:12364").read().decode()
        'response'
        >>> httpd = httpd.stop()
        >>>
        >>> with ThreadedHTTPServer(("localhost", 12364), RequestHandler) as httpd:
        ...     request.urlopen("http://localhost:12364").read().decode()
        'response'

    Args:
        server_address: The address on which the server is listening. Tuple
                        containing a string giving the address, and an integer
                        port number: ``('127.0.0.1', 80)``, for example.
        RequestHandlerClass: An instance of this class is created for each
                             request. Should provide methods like ``do_POST``
                             and ``do_GET`` to handle requests of the specific
                             types.
    """

    #: Allow socket address to be reused. Feature of
    #: :class:`~socketserver.TCPServer` which is the parent of
    #: :class:`~http.server.HTTPServer`.
    allow_reuse_address = True

    def __init__(
        self,
        server_address: Tuple[str, int],
        RequestHandlerClass: Type[BaseHTTPRequestHandler],
    ):
        # mypy complains about `bind_and_activate` being an unknown argument
        # to HTTPServer, which is weird, as HTTPServer inherits from TCPServer
        # without overloading __init__, while TCPServer does have the argument
        # with it also being documented. See python sources:
        # https://github.com/python/cpython/blob/8c19a44/Lib/http/server.py#L130
        # https://github.com/python/cpython/blob/cebe9ee/Lib/socketserver.py#L394
        HTTPServer.__init__(  # type: ignore
            self,
            server_address=server_address,
            RequestHandlerClass=RequestHandlerClass,
            bind_and_activate=False,
        )
        self._thread: Optional[StoppableThread] = None

    def __enter__(self: THS) -> THS:
        """Starts the threaded server when entering a context."""
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        """Joins the threaded server when leaving a context."""
        self.stop()
        self._thread.join()

    def __del__(self):
        """Stops the threaded server on deletion."""
        if not self.stopped():
            self.stop()

    def start(self: THS) -> THS:
        """(Re)creates the socket, binds and activates it, starts listening."""
        if not self.stopped():
            return self
        # (Re)create, bind and activate socket.
        self.socket = socket(self.address_family, self.socket_type)
        self.server_bind()
        self.server_activate()
        # Starts `serve_forever` loop in its own thread.
        self._thread = StoppableThread(
            target=self.serve_forever, name="ThreadedHTTPServer", daemon=True
        )
        self._thread.start()
        return self

    def stop(self: THS) -> THS:
        """Closes the socket and server. Can be re-opened using :meth:`start`."""
        if self.stopped():
            return self
        assert self._thread is not None  # satisfy mypy for next loc
        self._thread.stop()
        self.shutdown()
        self.server_close()
        self.socket.close()
        return self

    def stopped(self) -> bool:
        """Returns whether the server is currently active."""
        return self._thread is None or not self._thread.is_alive()
