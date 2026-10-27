/**
 * Attaches the CSRF token (from <meta name="csrf-token">) as an
 * X-CSRFToken header on every same-origin fetch() that doesn't already set
 * one, so every existing and future fetch() POST call is covered without
 * having to edit each call site individually. Flask-WTF's CSRFProtect
 * accepts the token from this header.
 */
(function () {
    const meta = document.querySelector('meta[name="csrf-token"]');
    if (!meta) return;
    const token = meta.getAttribute('content');

    const originalFetch = window.fetch;
    window.fetch = function (input, init) {
        init = init || {};
        const method = (init.method || 'GET').toUpperCase();
        if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {
            const url = typeof input === 'string' ? input : input.url;
            const isSameOrigin = url.startsWith('/') || url.startsWith(window.location.origin);
            if (isSameOrigin) {
                init.headers = init.headers || {};
                if (init.headers instanceof Headers) {
                    if (!init.headers.has('X-CSRFToken')) {
                        init.headers.set('X-CSRFToken', token);
                    }
                } else if (!('X-CSRFToken' in init.headers)) {
                    init.headers['X-CSRFToken'] = token;
                }
            }
        }
        return originalFetch(input, init);
    };
})();
